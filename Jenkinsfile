triggers {
    githubPullRequest(
        events: [
            [$class: 'GitHubPRPushEvent'],
            [$class: 'GitHubPRCommentEvent']
        ],
        cancelQueued: false
    )
}

stages {
    stage('Checkout PR') {
        steps {
            checkout([
                $class: 'GitSCM',
                branches: [[name: '${sha1}']],
                extensions: [
                    [$class: 'CloneOption', depth: 1, shallow: true]
                ],
                userRemoteConfigs: [[
                    url: 'https://github.com/Camille0512/NLA.git',
                    credentialsId: 'github-token',
                    refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
                ]]
            ])
        }
    }

    stage('Build & Test') {
        steps {
            sh 'pytest lu_decomposition_test.py'
            junit '**/JenkinsLogs/surefire-reports/*.xml'
        }
    }
}

post {
    always {
        cleanWs()  // Clean workspace after build
    }
}