properties([
    pipelineTriggers([githubPush()])
//     pipelineTriggers([githubPullRequests()])
])

// // Not figured out yet
// properties([
//     pipelineTriggers([
//         [
//             $class: 'GitHubPRTrigger',
//             events: [
//                 [$class: 'GitHubPROpenEvent']
//             ]
//         ]
//     ])
// ])

pipeline {
    agent any
    tools {
        git 'Default'  // Uses the Git tool configured in Jenkins Global Tools
    }
    environment {
        GIT_BRANCH = 'refs/heads/develop'
        GIT_URL = 'https://github.com/Camille0512/NLA.git'
        JENKINS_CREDENTIAL_ID = 'jenkins_nla'
        JENKINS_LOG = sh(script: 'JENKINS_LOG', returnStdout: true).trim()
        DATETIME = sh(script: 'date +"%Y%m%d_%H%M%S"', returnStdout: true).trim()
    }
    stages {
        stage('Checkout PR') {
            steps {
                sh 'echo "Start PR checkout"'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: GIT_BRANCH]], // ${sha1}
                    extensions: [
                        [$class: 'CloneOption', depth: 1, shallow: true]
                    ],
                    userRemoteConfigs: [[
                        url: GIT_URL,
                        credentialsId: JENKINS_CREDENTIAL_ID,
                        refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
                    ]]
                ])
                sh 'echo "Finish PR checkout"'
            }
        }

        stage('Build & Test') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip3 install pytest
                    python3 -m pytest --junitxml=${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_sample_test-results.xml
                    python3 -m pytest --junitxml=${JENKINS_LOG}/JenkinsLogs/surefire-reports/${DATETIME}_lu_decomposition_test-results.xml
                '''
                junit '**/JenkinsLogs/surefire-reports/*.xml'
                sh 'echo "Finish Build & Test"'
            }
        }
    }
}

node {
    cleanWs()  // Clean workspace after build
}